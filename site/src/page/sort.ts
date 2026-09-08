/**
 * Index-table sorting, bundled by esbuild into assets/sort.js.
 *
 * Vanilla progressive enhancement: the table is fully rendered statically;
 * this only reorders existing rows. Cells opt into a sort value via
 * data-sort-value (numeric when the header has data-sort-type="number");
 * missing values always sort last.
 */

function cellKey(row: HTMLTableRowElement, index: number): string {
  const cell = row.cells[index];
  if (cell === undefined) {
    return '';
  }
  return cell.dataset['sortValue'] ?? cell.textContent?.trim() ?? '';
}

function compareRows(
  a: HTMLTableRowElement,
  b: HTMLTableRowElement,
  index: number,
  numeric: boolean,
  ascending: boolean,
): number {
  const ka = cellKey(a, index);
  const kb = cellKey(b, index);
  if (ka === '' || kb === '') {
    // Missing values sort last regardless of direction.
    if (ka === '' && kb === '') {
      return 0;
    }
    return ka === '' ? 1 : -1;
  }
  let result: number;
  if (numeric) {
    result = Number.parseFloat(ka) - Number.parseFloat(kb);
  } else {
    result = ka < kb ? -1 : ka > kb ? 1 : 0;
  }
  return ascending ? result : -result;
}

function wireTable(table: HTMLTableElement): void {
  const body = table.tBodies[0];
  if (body === undefined) {
    return;
  }
  const headers = Array.from(table.querySelectorAll<HTMLTableCellElement>('thead th'));
  headers.forEach((header, index) => {
    header.tabIndex = 0;
    const activate = (): void => {
      const ascending = header.getAttribute('aria-sort') !== 'ascending';
      for (const other of headers) {
        other.removeAttribute('aria-sort');
      }
      header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
      const numeric = header.dataset['sortType'] === 'number';
      const rows = Array.from(body.rows);
      rows.sort((a, b) => compareRows(a, b, index, numeric, ascending));
      for (const row of rows) {
        body.appendChild(row);
      }
    };
    header.addEventListener('click', activate);
    header.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate();
      }
    });
  });
}

for (const table of document.querySelectorAll<HTMLTableElement>('table.sortable')) {
  wireTable(table);
}
